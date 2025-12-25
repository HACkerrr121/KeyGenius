package Backend;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.File;
import java.util.stream.Stream;
import org.audiveris.omr.sheet.Book;
import org.audiveris.omr.sheet.Sheet;
import org.audiveris.omr.sheet.SheetStub;
import org.audiveris.omr.sheet.SystemInfo;
import org.audiveris.omr.sig.inter.Inter;
import org.audiveris.omr.sig.SIGraph;
import org.audiveris.omr.sig.inter.HeadInter;
import org.audiveris.omr.*;
import java.awt.Rectangle;

public class CoordinateFinder {
    
    private String[] imageLabels;
    private String outPath;
    private File directory;
    private String inputFolder;

    public CoordinateFinder(String outPath, String inputFolder) {
        this.inputFolder = inputFolder;
        this.outPath = outPath;
        this.directory = new File(inputFolder);
        System.out.println(this.directory.isDirectory());
        imageLabels = directory.list();
    }

    public String[] getImageLabels() {
        return imageLabels;
    }
    
    public int getLength() {
        return imageLabels.length;
    }

    public void exportCoordinates() {
        try (PrintWriter writer = new PrintWriter(outPath + "/coordinates.csv")) {
            writer.println("image,x,y");
    
            String[] coords = new String[getLength()];
            for (String name : imageLabels) {
                Path imagePath = Paths.get(inputFolder, name);
                Book book = Book.loadBook(imagePath);
                book.transcribe(null, null, false);
                int num = 0;
                for (SheetStub stub : book.getStubs()) {
                    Sheet sheet = stub.getSheet();
                    for(SystemInfo system: sheet.getSystems()) {
                        SIGraph sig = system.getSig();
                        
                        for (Inter inter : sig.inters(HeadInter.class)) {
                            HeadInter head = (HeadInter) inter;
                            Rectangle bounds = head.getBounds();
                            int x = (int)(bounds.getX() + bounds.getWidth()/2);
                            int y = (int)(bounds.getY() + bounds.getHeight()/2);
                            writer.println(name + "," + x + "," + y);
                        }
                    }
                }
                book.close(0);
            }         
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
 
    public static void main(String[] args) {
        String base = "/Users/anandkashyap/Documents/GitHub/KeyGenius/Backend/Music_Data";
        CoordinateFinder hi = new CoordinateFinder(base, base + "/Scores");
        System.out.println(hi.getImageLabels()[0]);
        hi.exportCoordinates();
    }

}
